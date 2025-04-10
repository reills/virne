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
  id 687
  arrival_time 14558.669214189824
  lifetime 116.02496729600259
  num_nodes 15
  type "path"
  node [
    id 0
    label "0"
    cpu 40
    gpu 17
    rom 23
  ]
  node [
    id 1
    label "1"
    cpu 30
    gpu 40
    rom 41
  ]
  node [
    id 2
    label "2"
    cpu 17
    gpu 4
    rom 15
  ]
  node [
    id 3
    label "3"
    cpu 29
    gpu 14
    rom 46
  ]
  node [
    id 4
    label "4"
    cpu 8
    gpu 20
    rom 19
  ]
  node [
    id 5
    label "5"
    cpu 24
    gpu 13
    rom 27
  ]
  node [
    id 6
    label "6"
    cpu 16
    gpu 43
    rom 9
  ]
  node [
    id 7
    label "7"
    cpu 28
    gpu 23
    rom 5
  ]
  node [
    id 8
    label "8"
    cpu 44
    gpu 20
    rom 27
  ]
  node [
    id 9
    label "9"
    cpu 31
    gpu 25
    rom 9
  ]
  node [
    id 10
    label "10"
    cpu 0
    gpu 5
    rom 13
  ]
  node [
    id 11
    label "11"
    cpu 43
    gpu 34
    rom 17
  ]
  node [
    id 12
    label "12"
    cpu 17
    gpu 50
    rom 34
  ]
  node [
    id 13
    label "13"
    cpu 18
    gpu 5
    rom 48
  ]
  node [
    id 14
    label "14"
    cpu 31
    gpu 22
    rom 42
  ]
  edge [
    source 0
    target 1
    bw 9
  ]
  edge [
    source 1
    target 2
    bw 27
  ]
  edge [
    source 2
    target 3
    bw 7
  ]
  edge [
    source 3
    target 4
    bw 29
  ]
  edge [
    source 4
    target 5
    bw 43
  ]
  edge [
    source 5
    target 6
    bw 10
  ]
  edge [
    source 6
    target 7
    bw 20
  ]
  edge [
    source 7
    target 8
    bw 46
  ]
  edge [
    source 8
    target 9
    bw 31
  ]
  edge [
    source 9
    target 10
    bw 16
  ]
  edge [
    source 10
    target 11
    bw 27
  ]
  edge [
    source 11
    target 12
    bw 44
  ]
  edge [
    source 12
    target 13
    bw 7
  ]
  edge [
    source 13
    target 14
    bw 15
  ]
]
