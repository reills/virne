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
  id 254
  arrival_time 4857.516567601132
  lifetime 482.59537230942766
  num_nodes 13
  type "path"
  node [
    id 0
    label "0"
    cpu 5
    gpu 5
    rom 5
  ]
  node [
    id 1
    label "1"
    cpu 15
    gpu 19
    rom 17
  ]
  node [
    id 2
    label "2"
    cpu 32
    gpu 44
    rom 50
  ]
  node [
    id 3
    label "3"
    cpu 47
    gpu 39
    rom 37
  ]
  node [
    id 4
    label "4"
    cpu 21
    gpu 1
    rom 39
  ]
  node [
    id 5
    label "5"
    cpu 50
    gpu 43
    rom 39
  ]
  node [
    id 6
    label "6"
    cpu 44
    gpu 41
    rom 49
  ]
  node [
    id 7
    label "7"
    cpu 21
    gpu 7
    rom 33
  ]
  node [
    id 8
    label "8"
    cpu 42
    gpu 39
    rom 45
  ]
  node [
    id 9
    label "9"
    cpu 12
    gpu 45
    rom 6
  ]
  node [
    id 10
    label "10"
    cpu 23
    gpu 39
    rom 30
  ]
  node [
    id 11
    label "11"
    cpu 14
    gpu 31
    rom 42
  ]
  node [
    id 12
    label "12"
    cpu 19
    gpu 31
    rom 31
  ]
  edge [
    source 0
    target 1
    bw 49
  ]
  edge [
    source 1
    target 2
    bw 5
  ]
  edge [
    source 2
    target 3
    bw 26
  ]
  edge [
    source 3
    target 4
    bw 23
  ]
  edge [
    source 4
    target 5
    bw 27
  ]
  edge [
    source 5
    target 6
    bw 14
  ]
  edge [
    source 6
    target 7
    bw 12
  ]
  edge [
    source 7
    target 8
    bw 19
  ]
  edge [
    source 8
    target 9
    bw 42
  ]
  edge [
    source 9
    target 10
    bw 50
  ]
  edge [
    source 10
    target 11
    bw 24
  ]
  edge [
    source 11
    target 12
    bw 5
  ]
]
