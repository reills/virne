graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "gpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "rom"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  link_attrs_setting "_networkx_list_start"
  link_attrs_setting [
    name "bw"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "link"
    type "resource"
  ]
  id 272
  arrival_time 5259.108641577308
  lifetime 183.8058617106651
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 4
    gpu 48
    rom 25
  ]
  node [
    id 1
    label "1"
    cpu 3
    gpu 46
    rom 26
  ]
  node [
    id 2
    label "2"
    cpu 48
    gpu 34
    rom 33
  ]
  node [
    id 3
    label "3"
    cpu 2
    gpu 50
    rom 31
  ]
  node [
    id 4
    label "4"
    cpu 20
    gpu 16
    rom 10
  ]
  node [
    id 5
    label "5"
    cpu 47
    gpu 28
    rom 0
  ]
  node [
    id 6
    label "6"
    cpu 32
    gpu 34
    rom 41
  ]
  node [
    id 7
    label "7"
    cpu 14
    gpu 1
    rom 36
  ]
  node [
    id 8
    label "8"
    cpu 26
    gpu 15
    rom 46
  ]
  node [
    id 9
    label "9"
    cpu 42
    gpu 33
    rom 36
  ]
  edge [
    source 0
    target 1
    bw 18
  ]
  edge [
    source 1
    target 2
    bw 3
  ]
  edge [
    source 2
    target 3
    bw 15
  ]
  edge [
    source 3
    target 4
    bw 26
  ]
  edge [
    source 4
    target 5
    bw 47
  ]
  edge [
    source 5
    target 6
    bw 39
  ]
  edge [
    source 6
    target 7
    bw 17
  ]
  edge [
    source 7
    target 8
    bw 16
  ]
  edge [
    source 8
    target 9
    bw 43
  ]
]
